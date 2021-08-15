/**
 */
package slco.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import slco.Port;
import slco.SlcoPackage;
import slco.UnidirectionalChannel;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Unidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco.impl.UnidirectionalChannelImpl#getSourcePort <em>Source Port</em>}</li>
 *   <li>{@link slco.impl.UnidirectionalChannelImpl#getTargetPort <em>Target Port</em>}</li>
 *   <li>{@link slco.impl.UnidirectionalChannelImpl#getSourceObject <em>Source Object</em>}</li>
 *   <li>{@link slco.impl.UnidirectionalChannelImpl#getTargetObject <em>Target Object</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class UnidirectionalChannelImpl extends ChannelImpl implements UnidirectionalChannel {
	/**
	 * The cached value of the '{@link #getSourcePort() <em>Source Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSourcePort()
	 * @generated
	 * @ordered
	 */
	protected Port sourcePort;

	/**
	 * The cached value of the '{@link #getTargetPort() <em>Target Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTargetPort()
	 * @generated
	 * @ordered
	 */
	protected Port targetPort;

	/**
	 * The cached value of the '{@link #getSourceObject() <em>Source Object</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSourceObject()
	 * @generated
	 * @ordered
	 */
	protected slco.Object sourceObject;

	/**
	 * The cached value of the '{@link #getTargetObject() <em>Target Object</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTargetObject()
	 * @generated
	 * @ordered
	 */
	protected slco.Object targetObject;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected UnidirectionalChannelImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return SlcoPackage.Literals.UNIDIRECTIONAL_CHANNEL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port getSourcePort() {
		if (sourcePort != null && sourcePort.eIsProxy()) {
			InternalEObject oldSourcePort = (InternalEObject)sourcePort;
			sourcePort = (Port)eResolveProxy(oldSourcePort);
			if (sourcePort != oldSourcePort) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT, oldSourcePort, sourcePort));
			}
		}
		return sourcePort;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port basicGetSourcePort() {
		return sourcePort;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSourcePort(Port newSourcePort) {
		Port oldSourcePort = sourcePort;
		sourcePort = newSourcePort;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT, oldSourcePort, sourcePort));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port getTargetPort() {
		if (targetPort != null && targetPort.eIsProxy()) {
			InternalEObject oldTargetPort = (InternalEObject)targetPort;
			targetPort = (Port)eResolveProxy(oldTargetPort);
			if (targetPort != oldTargetPort) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT, oldTargetPort, targetPort));
			}
		}
		return targetPort;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port basicGetTargetPort() {
		return targetPort;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTargetPort(Port newTargetPort) {
		Port oldTargetPort = targetPort;
		targetPort = newTargetPort;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT, oldTargetPort, targetPort));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object getSourceObject() {
		if (sourceObject != null && sourceObject.eIsProxy()) {
			InternalEObject oldSourceObject = (InternalEObject)sourceObject;
			sourceObject = (slco.Object)eResolveProxy(oldSourceObject);
			if (sourceObject != oldSourceObject) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT, oldSourceObject, sourceObject));
			}
		}
		return sourceObject;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object basicGetSourceObject() {
		return sourceObject;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSourceObject(slco.Object newSourceObject) {
		slco.Object oldSourceObject = sourceObject;
		sourceObject = newSourceObject;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT, oldSourceObject, sourceObject));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object getTargetObject() {
		if (targetObject != null && targetObject.eIsProxy()) {
			InternalEObject oldTargetObject = (InternalEObject)targetObject;
			targetObject = (slco.Object)eResolveProxy(oldTargetObject);
			if (targetObject != oldTargetObject) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT, oldTargetObject, targetObject));
			}
		}
		return targetObject;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object basicGetTargetObject() {
		return targetObject;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTargetObject(slco.Object newTargetObject) {
		slco.Object oldTargetObject = targetObject;
		targetObject = newTargetObject;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT, oldTargetObject, targetObject));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
				if (resolve) return getSourcePort();
				return basicGetSourcePort();
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
				if (resolve) return getTargetPort();
				return basicGetTargetPort();
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
				if (resolve) return getSourceObject();
				return basicGetSourceObject();
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
				if (resolve) return getTargetObject();
				return basicGetTargetObject();
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
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
				setSourcePort((Port)newValue);
				return;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
				setTargetPort((Port)newValue);
				return;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
				setSourceObject((slco.Object)newValue);
				return;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
				setTargetObject((slco.Object)newValue);
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
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
				setSourcePort((Port)null);
				return;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
				setTargetPort((Port)null);
				return;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
				setSourceObject((slco.Object)null);
				return;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
				setTargetObject((slco.Object)null);
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
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
				return sourcePort != null;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
				return targetPort != null;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
				return sourceObject != null;
			case SlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
				return targetObject != null;
		}
		return super.eIsSet(featureID);
	}

} //UnidirectionalChannelImpl
