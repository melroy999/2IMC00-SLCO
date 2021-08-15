/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.AcquireStatement;
import nqc.NqcPackage;
import nqc.Statement;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Acquire Statement</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.AcquireStatementImpl#getResource <em>Resource</em>}</li>
 *   <li>{@link nqc.impl.AcquireStatementImpl#getBody <em>Body</em>}</li>
 *   <li>{@link nqc.impl.AcquireStatementImpl#getHandler <em>Handler</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class AcquireStatementImpl extends StatementImpl implements AcquireStatement {
	/**
	 * The cached value of the '{@link #getResource() <em>Resource</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getResource()
	 * @generated
	 * @ordered
	 */
	protected AcquireStatement resource;

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
	protected AcquireStatementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getAcquireStatement();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AcquireStatement getResource() {
		return resource;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetResource(AcquireStatement newResource, NotificationChain msgs) {
		AcquireStatement oldResource = resource;
		resource = newResource;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.ACQUIRE_STATEMENT__RESOURCE, oldResource, newResource);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setResource(AcquireStatement newResource) {
		if (newResource != resource) {
			NotificationChain msgs = null;
			if (resource != null)
				msgs = ((InternalEObject)resource).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.ACQUIRE_STATEMENT__RESOURCE, null, msgs);
			if (newResource != null)
				msgs = ((InternalEObject)newResource).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.ACQUIRE_STATEMENT__RESOURCE, null, msgs);
			msgs = basicSetResource(newResource, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.ACQUIRE_STATEMENT__RESOURCE, newResource, newResource));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.ACQUIRE_STATEMENT__BODY, oldBody, newBody);
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
				msgs = ((InternalEObject)body).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.ACQUIRE_STATEMENT__BODY, null, msgs);
			if (newBody != null)
				msgs = ((InternalEObject)newBody).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.ACQUIRE_STATEMENT__BODY, null, msgs);
			msgs = basicSetBody(newBody, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.ACQUIRE_STATEMENT__BODY, newBody, newBody));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, NqcPackage.ACQUIRE_STATEMENT__HANDLER, oldHandler, newHandler);
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
				msgs = ((InternalEObject)handler).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - NqcPackage.ACQUIRE_STATEMENT__HANDLER, null, msgs);
			if (newHandler != null)
				msgs = ((InternalEObject)newHandler).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - NqcPackage.ACQUIRE_STATEMENT__HANDLER, null, msgs);
			msgs = basicSetHandler(newHandler, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.ACQUIRE_STATEMENT__HANDLER, newHandler, newHandler));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.ACQUIRE_STATEMENT__RESOURCE:
				return basicSetResource(null, msgs);
			case NqcPackage.ACQUIRE_STATEMENT__BODY:
				return basicSetBody(null, msgs);
			case NqcPackage.ACQUIRE_STATEMENT__HANDLER:
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
			case NqcPackage.ACQUIRE_STATEMENT__RESOURCE:
				return getResource();
			case NqcPackage.ACQUIRE_STATEMENT__BODY:
				return getBody();
			case NqcPackage.ACQUIRE_STATEMENT__HANDLER:
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
			case NqcPackage.ACQUIRE_STATEMENT__RESOURCE:
				setResource((AcquireStatement)newValue);
				return;
			case NqcPackage.ACQUIRE_STATEMENT__BODY:
				setBody((Statement)newValue);
				return;
			case NqcPackage.ACQUIRE_STATEMENT__HANDLER:
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
			case NqcPackage.ACQUIRE_STATEMENT__RESOURCE:
				setResource((AcquireStatement)null);
				return;
			case NqcPackage.ACQUIRE_STATEMENT__BODY:
				setBody((Statement)null);
				return;
			case NqcPackage.ACQUIRE_STATEMENT__HANDLER:
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
			case NqcPackage.ACQUIRE_STATEMENT__RESOURCE:
				return resource != null;
			case NqcPackage.ACQUIRE_STATEMENT__BODY:
				return body != null;
			case NqcPackage.ACQUIRE_STATEMENT__HANDLER:
				return handler != null;
		}
		return super.eIsSet(featureID);
	}

} //AcquireStatementImpl
