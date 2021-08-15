/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.EventTypeConstant;
import nqc.MonitorHandler;
import nqc.NqcPackage;
import nqc.Statement;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Monitor Handler</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.MonitorHandlerImpl#getCatchEvent <em>Catch Event</em>}</li>
 *   <li>{@link nqc.impl.MonitorHandlerImpl#getHandler <em>Handler</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class MonitorHandlerImpl extends EObjectImpl implements MonitorHandler {
	/**
	 * The cached value of the '{@link #getCatchEvent() <em>Catch Event</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCatchEvent()
	 * @generated
	 * @ordered
	 */
	protected EventTypeConstant catchEvent;

	/**
	 * The cached value of the '{@link #getHandler() <em>Handler</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getHandler()
	 * @generated
	 * @ordered
	 */
	protected Statement handler;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MonitorHandlerImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getMonitorHandler();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EventTypeConstant getCatchEvent() {
		return catchEvent;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetCatchEvent(EventTypeConstant newCatchEvent, NotificationChain msgs) {
		EventTypeConstant oldCatchEvent = catchEvent;
		catchEvent = newCatchEvent;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.MONITOR_HANDLER__CATCH_EVENT, oldCatchEvent, newCatchEvent);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCatchEvent(EventTypeConstant newCatchEvent) {
		if (newCatchEvent != catchEvent) {
			NotificationChain msgs = null;
			if (catchEvent != null)
				msgs = ((InternalEObject)catchEvent).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.MONITOR_HANDLER__CATCH_EVENT, null, msgs);
			if (newCatchEvent != null)
				msgs = ((InternalEObject)newCatchEvent).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.MONITOR_HANDLER__CATCH_EVENT, null, msgs);
			msgs = basicSetCatchEvent(newCatchEvent, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.MONITOR_HANDLER__CATCH_EVENT, newCatchEvent, newCatchEvent));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Statement getHandler() {
		return handler;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetHandler(Statement newHandler, NotificationChain msgs) {
		Statement oldHandler = handler;
		handler = newHandler;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.MONITOR_HANDLER__HANDLER, oldHandler, newHandler);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setHandler(Statement newHandler) {
		if (newHandler != handler) {
			NotificationChain msgs = null;
			if (handler != null)
				msgs = ((InternalEObject)handler).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.MONITOR_HANDLER__HANDLER, null, msgs);
			if (newHandler != null)
				msgs = ((InternalEObject)newHandler).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.MONITOR_HANDLER__HANDLER, null, msgs);
			msgs = basicSetHandler(newHandler, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.MONITOR_HANDLER__HANDLER, newHandler, newHandler));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.MONITOR_HANDLER__CATCH_EVENT:
				return basicSetCatchEvent(null, msgs);
			case NqcPackage.MONITOR_HANDLER__HANDLER:
				return basicSetHandler(null, msgs);
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
			case NqcPackage.MONITOR_HANDLER__CATCH_EVENT:
				return getCatchEvent();
			case NqcPackage.MONITOR_HANDLER__HANDLER:
				return getHandler();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case NqcPackage.MONITOR_HANDLER__CATCH_EVENT:
				setCatchEvent((EventTypeConstant)newValue);
				return;
			case NqcPackage.MONITOR_HANDLER__HANDLER:
				setHandler((Statement)newValue);
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
			case NqcPackage.MONITOR_HANDLER__CATCH_EVENT:
				setCatchEvent((EventTypeConstant)null);
				return;
			case NqcPackage.MONITOR_HANDLER__HANDLER:
				setHandler((Statement)null);
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
			case NqcPackage.MONITOR_HANDLER__CATCH_EVENT:
				return catchEvent != null;
			case NqcPackage.MONITOR_HANDLER__HANDLER:
				return handler != null;
		}
		return super.eIsSet(featureID);
	}

} //MonitorHandlerImpl
