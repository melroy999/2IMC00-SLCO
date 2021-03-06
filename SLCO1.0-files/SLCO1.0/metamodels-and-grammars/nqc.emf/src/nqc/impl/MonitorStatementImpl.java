/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import java.util.Collection;

import nqc.EventTypeConstant;
import nqc.MonitorHandler;
import nqc.MonitorStatement;
import nqc.NqcPackage;
import nqc.Statement;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Monitor Statement</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.MonitorStatementImpl#getEvents <em>Events</em>}</li>
 *   <li>{@link nqc.impl.MonitorStatementImpl#getBody <em>Body</em>}</li>
 *   <li>{@link nqc.impl.MonitorStatementImpl#getHandlers <em>Handlers</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class MonitorStatementImpl extends StatementImpl implements MonitorStatement {
	/**
	 * The cached value of the '{@link #getEvents() <em>Events</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getEvents()
	 * @generated
	 * @ordered
	 */
	protected EList<EventTypeConstant> events;

	/**
	 * The cached value of the '{@link #getBody() <em>Body</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBody()
	 * @generated
	 * @ordered
	 */
	protected Statement body;

	/**
	 * The cached value of the '{@link #getHandlers() <em>Handlers</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getHandlers()
	 * @generated
	 * @ordered
	 */
	protected EList<MonitorHandler> handlers;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MonitorStatementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getMonitorStatement();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<EventTypeConstant> getEvents() {
		if (events == null) {
			events = new EObjectContainmentEList<EventTypeConstant>(EventTypeConstant.class, this, NqcPackage.MONITOR_STATEMENT__EVENTS);
		}
		return events;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Statement getBody() {
		return body;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetBody(Statement newBody, NotificationChain msgs) {
		Statement oldBody = body;
		body = newBody;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.MONITOR_STATEMENT__BODY, oldBody, newBody);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setBody(Statement newBody) {
		if (newBody != body) {
			NotificationChain msgs = null;
			if (body != null)
				msgs = ((InternalEObject)body).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.MONITOR_STATEMENT__BODY, null, msgs);
			if (newBody != null)
				msgs = ((InternalEObject)newBody).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.MONITOR_STATEMENT__BODY, null, msgs);
			msgs = basicSetBody(newBody, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.MONITOR_STATEMENT__BODY, newBody, newBody));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<MonitorHandler> getHandlers() {
		if (handlers == null) {
			handlers = new EObjectContainmentEList<MonitorHandler>(MonitorHandler.class, this, NqcPackage.MONITOR_STATEMENT__HANDLERS);
		}
		return handlers;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.MONITOR_STATEMENT__EVENTS:
				return ((InternalEList<?>)getEvents()).basicRemove(otherEnd, msgs);
			case NqcPackage.MONITOR_STATEMENT__BODY:
				return basicSetBody(null, msgs);
			case NqcPackage.MONITOR_STATEMENT__HANDLERS:
				return ((InternalEList<?>)getHandlers()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.MONITOR_STATEMENT__EVENTS:
				return getEvents();
			case NqcPackage.MONITOR_STATEMENT__BODY:
				return getBody();
			case NqcPackage.MONITOR_STATEMENT__HANDLERS:
				return getHandlers();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case NqcPackage.MONITOR_STATEMENT__EVENTS:
				getEvents().clear();
				getEvents().addAll((Collection<? extends EventTypeConstant>)newValue);
				return;
			case NqcPackage.MONITOR_STATEMENT__BODY:
				setBody((Statement)newValue);
				return;
			case NqcPackage.MONITOR_STATEMENT__HANDLERS:
				getHandlers().clear();
				getHandlers().addAll((Collection<? extends MonitorHandler>)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case NqcPackage.MONITOR_STATEMENT__EVENTS:
				getEvents().clear();
				return;
			case NqcPackage.MONITOR_STATEMENT__BODY:
				setBody((Statement)null);
				return;
			case NqcPackage.MONITOR_STATEMENT__HANDLERS:
				getHandlers().clear();
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case NqcPackage.MONITOR_STATEMENT__EVENTS:
				return events != null && !events.isEmpty();
			case NqcPackage.MONITOR_STATEMENT__BODY:
				return body != null;
			case NqcPackage.MONITOR_STATEMENT__HANDLERS:
				return handlers != null && !handlers.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //MonitorStatementImpl
